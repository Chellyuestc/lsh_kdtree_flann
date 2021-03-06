/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/***********************************************************************
 * Author: Vincent Rabaud
 *************************************************************************/

#ifndef FLANN_LSH_TABLE_H_
#define FLANN_LSH_TABLE_H_

#include <algorithm>
#include <iostream>
#include <random>
#include <iomanip>
#include <limits.h>
// TODO as soon as we use C++0x, use the code in USE_UNORDERED_MAP
#if USE_UNORDERED_MAP
#include <unordered_map>
#else
#include <map>
#endif
#include <math.h>
#include <stddef.h>
#include <fstream>

#include "flann/flann.hpp"
#include "flann/util/dynamic_bitset.h"
#include "flann/util/matrix.h"

namespace flann
{

namespace lsh
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** What is stored in an LSH bucket
 */
typedef uint32_t FeatureIndex;
/** The id from which we can get a bucket back in an LSH table
 */
typedef std::vector<int>  BucketKey;

typedef std::vector<float>  BucketKey_float;

/** A bucket in an LSH table
 */
typedef std::vector<FeatureIndex> Bucket;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** POD for stats about an LSH table
 */
struct LshStats
{
    std::vector<unsigned int> bucket_sizes_;
    size_t n_buckets_;
    size_t bucket_size_mean_;
    size_t bucket_size_median_;
    size_t bucket_size_min_;
    size_t bucket_size_max_;
    size_t bucket_size_std_dev;
    /** Each contained vector contains three value: beginning/end for interval, number of elements in the bin
     */
    std::vector<std::vector<unsigned int> > size_histogram_;
};

/** Overload the << operator for LshStats
 * @param out the streams
 * @param stats the stats to display
 * @return the streams
 */
inline std::ostream& operator <<(std::ostream& out, const LshStats& stats)
{
    size_t w = 20;
    out << "Lsh Table Stats:\n" << std::setw(w) << std::setiosflags(std::ios::right) << "N buckets : "
    << stats.n_buckets_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "mean size : "
    << std::setiosflags(std::ios::left) << stats.bucket_size_mean_ << "\n" << std::setw(w)
    << std::setiosflags(std::ios::right) << "median size : " << stats.bucket_size_median_ << "\n" << std::setw(w)
    << std::setiosflags(std::ios::right) << "min size : " << std::setiosflags(std::ios::left)
    << stats.bucket_size_min_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "max size : "
    << std::setiosflags(std::ios::left) << stats.bucket_size_max_;

    // Display the histogram
    out << std::endl << std::setw(w) << std::setiosflags(std::ios::right) << "histogram : "
    << std::setiosflags(std::ios::left);
    for (std::vector<std::vector<unsigned int> >::const_iterator iterator = stats.size_histogram_.begin(), end =
             stats.size_histogram_.end(); iterator != end; ++iterator) out << (*iterator)[0] << "-" << (*iterator)[1] << ": " << (*iterator)[2] << ",  ";

    return out;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Lsh hash table. As its key is a sub-feature, and as usually
 * the size of it is pretty small, we keep it as a continuous memory array.
 * The value is an index in the corpus of features (we keep it as an unsigned
 * int for pure memory reasons, it could be a size_t)
 */
template<typename ElementType>
class LshTable
{
public:
    /** A container of all the feature indices. Optimized for space
     */
#if USE_UNORDERED_MAP
    typedef std::unordered_map<BucketKey, Bucket> BucketsSpace;
#else
    typedef std::map<BucketKey, Bucket> BucketsSpace;
#endif

    /** A container of all the feature indices. Optimized for speed
     */
    typedef std::vector<Bucket> BucketsSpeed;

    /** Default constructor
     */
    LshTable()
    {
    }

    /** Default constructor
     * Create the mask and allocate the memory
     * @param feature_size is the size of the feature (considered as a ElementType[])
     * @param key_size is the number of bits that are turned on in the feature
     */
    LshTable(unsigned int /*feature_size*/, unsigned int /*key_size*/)
    {
        std::cerr << "LSH is not implemented for that type" << std::endl;
        throw;
    }

    /** Add a feature to the table
     * @param value the value to store for that feature
     * @param feature the feature itself
     */
    void add(unsigned int value, const ElementType* feature)
    {
        // Add the value to the corresponding bucket
        BucketKey_float key_float = getKey(feature);
		BucketKey key(key_size_);
		for (int i = 0; i < key_size_; i++)
			key[i] = floor(key_float[i]);
		/*
		ofstream RecordFile;
		RecordFile.open("LSH_buckets.txt", ios::app);
		for (int i = 0; i < key_size_; i++)
			RecordFile << key[i] << ' ';
		RecordFile << std::endl;
		*/
        switch (speed_level_) {
        case kArray:
            // That means we get the buckets from an array
			//put it here temporaryly
            //buckets_speed_[key].push_back(value);
			throw;
            break;
        case kBitsetHash:
            // That means we can check the bitset for the presence of a key
			//put it here temporary
            //key_bitset_.set(key);
            //buckets_space_[key].push_back(value);
			throw;
            break;
        case kHash:
        {
            // That means we have to check for the hash table for the presence of a key
            buckets_space_[key].push_back(value);
            break;
        }
        }
    }

	void add(unsigned int value, const ElementType* feature, unsigned int vec_len)
	{
		// Add the value to the corresponding bucket
		BucketKey_float key_float = getKey(feature);
		BucketKey key(key_size_);
		for (int i = 0; i < key_size_; i++)
			key[i] = floor(key_float[i]);

	}


    /** Add a set of features to the table
     * @param dataset the values to store
     */
	
    void add(const std::vector< std::pair<size_t, ElementType*> >& features)
    {
#if USE_UNORDERED_MAP
        buckets_space_.rehash((buckets_space_.size() + features.size()) * 1.2);
#endif
        // Add the features to the table
        for (size_t i = 0; i < features.size(); ++i) {
        	add(features[i].first, features[i].second);
        }
        // Now that the table is full, optimize it for speed/space
        optimize();
    }
	
    /** Get a bucket given the key
     * @param key
     * @return
     */
    inline const Bucket* getBucketFromKey(BucketKey key) const
    {
        // Generate other buckets
        switch (speed_level_) {
        case kArray:
            // That means we get the buckets from an array
			//put it here temporaryly
            //return &buckets_speed_[key];
			throw;
            break;
        case kBitsetHash:
            // That means we can check the bitset for the presence of a key
			//put it here temporaryly
            //if (key_bitset_.test(key)) return &buckets_space_.find(key)->second;
            //else return 0;
			throw;
            break;
        case kHash:
        {
            // That means we have to check for the hash table for the presence of a key
            BucketsSpace::const_iterator bucket_it, bucket_end = buckets_space_.end();
            bucket_it = buckets_space_.find(key);
            // Stop here if that bucket does not exist
            if (bucket_it == bucket_end) return 0;
            else return &bucket_it->second;
            break;
        }
        }
        return 0;
    }

    /** Compute the sub-signature of a feature
     */
	std::vector<float> getKey(const ElementType* /*feature*/) const
    {
        //std::cerr << "LSH is not implemented for that type" << std::endl;
		std::cerr << "This changed LSH is not implemented for unsigned char" << std::endl;
        throw;
        return 1;
    }
	

    /** Get statistics about the table
     * @return
     */
    LshStats getStats() const;


	long usedMemory()
	{
		return buckets_space_.size();
	}

	int features_in_a_bucket()
	{
		BucketsSpace::const_iterator bucket_begin = buckets_space_.begin(), bucket_end = buckets_space_.end();

		for (; bucket_begin != bucket_end; bucket_begin++)
		{
			std::cout << bucket_begin->second.size() << std::endl;
		}
		
		return 2;
	}

private:
    /** defines the speed fo the implementation
     * kArray uses a vector for storing data
     * kBitsetHash uses a hash map but checks for the validity of a key with a bitset
     * kHash uses a hash map only
     */
    enum SpeedLevel
    {
		//I only used the kHash manner
        kArray, kBitsetHash, kHash
    };

    /** Initialize some variables
     */
    void initialize(size_t key_size)
    {
        speed_level_ = kHash;
        key_size_ = key_size;
    }

    /** Optimize the table for speed/space
     */
    void optimize()
    {
		/*Do nothing here*/

		/*put it here temporaryly*/
		/*
        // If we are already using the fast storage, no need to do anything
		if (speed_level_ == kArray)
		{
			return;
		}
		
        // Use an array if it will be more than half full
        if (buckets_space_.size() > (unsigned int)((1 << key_size_) / 2)) {
            speed_level_ = kArray;
            // Fill the array version of it
            buckets_speed_.resize(1 << key_size_);
			for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket)
			{
				BucketIDKeyMap[key_bucket->first] = BucketIDTemp;
				BucketIDTemp++;
				buckets_speed_[BucketIDTemp] = key_bucket->second;
				//buckets_speed_[key_bucket->first] = key_bucket->second;
			}

            // Empty the hash table
            buckets_space_.clear();
            return;
        }
		
        // If the bitset is going to use less than 10% of the RAM of the hash map (at least 1 size_t for the key and two
        // for the vector) or less than 512MB (key_size_ <= 30)
        if (((std::max(buckets_space_.size(), buckets_speed_.size()) * CHAR_BIT * 3 * sizeof(BucketKey)) / 10
             >= size_t(1 << key_size_)) || (key_size_ <= 32)) {
            speed_level_ = kBitsetHash;
            key_bitset_.resize(1 << key_size_);
            key_bitset_.reset();
            // Try with the BucketsSpace
			for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket)
			{
				//key_bitset_.set(key_bucket->first);
			}
        }
        else {
            speed_level_ = kHash;
            key_bitset_.clear();
        }
		*/
    }

    template<typename Archive>
    void serialize(Archive& ar)
    {
    	int val;
    	if (Archive::is_saving::value) {
    		val = (int)speed_level_;
    	}
    	ar & val;
    	if (Archive::is_loading::value) {
    		speed_level_ = (SpeedLevel) val;
    	}

    	ar & key_size_;
    	ar & mask_;

    	if (speed_level_==kArray) {
    		ar & buckets_speed_;
    	}
    	if (speed_level_==kBitsetHash || speed_level_==kHash) {
    		ar & buckets_space_;
    	}
		if (speed_level_==kBitsetHash) {
			ar & key_bitset_;
		}
    }
    friend struct serialization::access;

    /** The vector of all the buckets if they are held for speed
     */
    BucketsSpeed buckets_speed_;

    /** The hash table of all the buckets in case we cannot use the speed version
     */
    BucketsSpace buckets_space_;

    /** What is used to store the data */
    SpeedLevel speed_level_;

    /** If the subkey is small enough, it will keep track of which subkeys are set through that bitset
     * That is just a speedup so that we don't look in the hash table (which can be mush slower that checking a bitset)
     */
    DynamicBitset key_bitset_;

    /** The size of the sub-signature in bits
     */
    unsigned int key_size_;

    // Members only used for the unsigned char specialization
    /** The mask to apply to a feature to get the hash key
     * Only used in the unsigned char case
     */
    std::vector<size_t> mask_;

	std::vector<size_t> probe_;

	Matrix<float> Hash_Matrix_;

	float Hash_Bias_;

	int vec_len;

	float Hash_W_;

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<>
inline LshTable<float>::LshTable(unsigned int feature_size, unsigned int subsignature_size)
{
	/*Init hash table*/
	initialize(subsignature_size);

	int M = subsignature_size;
	Hash_W_ = 1000;

	vec_len = feature_size;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> a_norm(0.0, 1.0);
	std::uniform_real_distribution<> b_unif(0.0, Hash_W_);

	Hash_Matrix_ = Matrix<float>(new float[vec_len * M], vec_len, M);

	for (int j = 0; j < M; j++)
	{
		for (int k = 0; k < vec_len; k++)
		{
			*(Hash_Matrix_[j] + k) = a_norm(gen);
		}
	}
	Hash_Bias_ = b_unif(gen);
}


/** Return the Subsignature of a float feature
* @param feature the feature to analyze
*/
template<>
inline std::vector<float> LshTable<float>::getKey(const float* feature) const
{
	const float* feature_block_ptr = feature;

	std::vector<float> subsignature(key_size_);
	
	float sum_upper = 0;

	for (int j = 0; j < key_size_; j++)
	{
		sum_upper = 0;
		for (int k = 0; k < vec_len; k++)
		{
			//std::cout << *(Hash_Matrix_[j] + k) << ' ' << *(feature_block_ptr + k) << std::endl;
			sum_upper += *(Hash_Matrix_[j] + k) * *(feature_block_ptr + k);
		}
		sum_upper += Hash_Bias_;
		subsignature[j] = sum_upper / Hash_W_;
	}

	return subsignature;
}

/*We will not use it here*/
/*
template<>
inline LshStats LshTable<unsigned char>::getStats() const
{
    LshStats stats;
    stats.bucket_size_mean_ = 0;
    if ((buckets_speed_.empty()) && (buckets_space_.empty())) {
        stats.n_buckets_ = 0;
        stats.bucket_size_median_ = 0;
        stats.bucket_size_min_ = 0;
        stats.bucket_size_max_ = 0;
        return stats;
    }

    if (!buckets_speed_.empty()) {
        for (BucketsSpeed::const_iterator pbucket = buckets_speed_.begin(); pbucket != buckets_speed_.end(); ++pbucket) {
            stats.bucket_sizes_.push_back(pbucket->size());
            stats.bucket_size_mean_ += pbucket->size();
        }
        stats.bucket_size_mean_ /= buckets_speed_.size();
        stats.n_buckets_ = buckets_speed_.size();
    }
    else {
        for (BucketsSpace::const_iterator x = buckets_space_.begin(); x != buckets_space_.end(); ++x) {
            stats.bucket_sizes_.push_back(x->second.size());
            stats.bucket_size_mean_ += x->second.size();
        }
        stats.bucket_size_mean_ /= buckets_space_.size();
        stats.n_buckets_ = buckets_space_.size();
    }

    std::sort(stats.bucket_sizes_.begin(), stats.bucket_sizes_.end());

    //  BOOST_FOREACH(int size, stats.bucket_sizes_)
    //          std::cout << size << " ";
    //  std::cout << std::endl;
    stats.bucket_size_median_ = stats.bucket_sizes_[stats.bucket_sizes_.size() / 2];
    stats.bucket_size_min_ = stats.bucket_sizes_.front();
    stats.bucket_size_max_ = stats.bucket_sizes_.back();

    // TODO compute mean and std
    //float mean, stddev;
    //stats.bucket_size_mean_ = mean;
    //stats.bucket_size_std_dev = stddev;

    // Include a histogram of the buckets
    unsigned int bin_start = 0;
    unsigned int bin_end = 20;
    bool is_new_bin = true;
    for (std::vector<unsigned int>::iterator iterator = stats.bucket_sizes_.begin(), end = stats.bucket_sizes_.end(); iterator
         != end; )
        if (*iterator < bin_end) {
            if (is_new_bin) {
                stats.size_histogram_.push_back(std::vector<unsigned int>(3, 0));
                stats.size_histogram_.back()[0] = bin_start;
                stats.size_histogram_.back()[1] = bin_end - 1;
                is_new_bin = false;
            }
            ++stats.size_histogram_.back()[2];
            ++iterator;
        }
        else {
            bin_start += 20;
            bin_end += 20;
            is_new_bin = true;
        }

    return stats;
}
*/
// End the two namespaces
}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* FLANN_LSH_TABLE_H_ */
